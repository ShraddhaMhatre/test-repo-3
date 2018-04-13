Lorem ipsum dolor sit amet, ad eos aliquam partiendo, mei aliquid tincidunt deterruisset et. Causae persequeris reprehendunt te eos. Volutpat reprehendunt et his. Eum ea tempor regione senserit, cu est convenire posidonium. Ea assum nonumy ius, vim repudiandae signiferumque conclusionemque at, per paulo eirmod at. Suscipit elaboraret delicatissimi ei mel.

Id ius legimus detraxit omittantur, graeci meliore percipit in duo, cum in nulla praesent. Id vim falli eruditi omittantur, no quis decore feugiat his. Quo ea ubique mollis ceteros, mel ornatus feugait salutatus eu, ei quas atomorum oportere eos. Eum quem laudem et, fugit inimicus abhorreant vim ei. Est numquam fabulas patrioque ut.

Ad mei alia dicant cotidieque. Cibo libris pericula vel in, enim lobortis in vis. Eros graece nam ei. Vim cu iudicabit evertitur intellegat, justo summo prodesset mel ea, cu aliquip volutpat reformidans cum. Autem exerci adipiscing at sea, tamquam nominati repudiandae ut nam, usu unum eius at.

Congue possim fierent pro in. In vix case tation primis. Sed dicit officiis ei, ut nec solum graeci. Altera equidem gubergren mel eu, usu dolores volutpat interesset id.

Eos id novum consulatu dissentiunt, sea ne solet iudicabit, pri ei natum recteque. Tation deseruisse scribentur at eam, patrioque definiebas ea vix. Vix mentitum definitiones ne, facer placerat philosophia eos et, id aeque phaedrum vim. Eos quas utinam discere ei.

Dolor singulis usu et. Solum tacimates eum ei, nibh commodo epicuri eum ne. Ne pri vocent reprimique, ad eos erroribus percipitur mediocritatem. Repudiare evertitur quo id. Dico sonet ei eos, id ridens nostrum vel, sit ornatus similique et.

Consul quodsi voluptatum ius in, eam eu oblique postulant referrentur, partiendo iudicabit eos id. At nam vidisse habemus, antiopam evertitur vituperatoribus ea pro. Minim semper maiorum no mei, duo inani dolorem accommodare id. Id nominati praesent nec, sit oblique luptatum dignissim cu, ornatus dolores has te. Affert impedit eos id, vide nihil te quo. Sea et vidisse referrentur, pri ea erant volumus legendos. Sit vivendo insolens democritum eu, pro dico equidem ex.

Ei summo liber iudico per, nec cu accumsan senserit sadipscing. Suscipiantur deterruisset per eu. At mutat nemore neglegentur eum, in has melius liberavisse, ex quis fabellas democritum ius. Cibo falli eu ius. Ea magna habemus fastidii mea, consequat temporibus nec ei.

Eius dolor te pri, ut sed erant putant. Agam sadipscing ea eos. Eu eam inani tractatos. Ei lorem error aliquam pro, wisi adhuc ludus ne sit. An sed quot nulla facilis, quo ei nusquam electram explicari.

Molestie incorrupte omittantur qui ut, decore soleat adversarium sea te. Ferri vitae scripserit at mea, an wisi definitiones vel. Duo labitur patrioque te, patrioque adversarium disputationi no pri. Vix cu tantas imperdiet, et duo odio modo adolescens. Laboramus prodesset interesset usu id, id eos dicant expetendis. Modo explicari te mei, malis iusto pro ex, ea scaevola officiis eam

int main()
{
   int n, i = 3, count, c;
 
   printf("Enter the number of prime numbers required\n");
   scanf("%d",&n);
 
   if ( n >= 1 )
   {
      printf("First %d prime numbers are :\n",n);
      printf("2\n");
   }
 
   for ( count = 2 ; count <= n ;  )
   {
      for ( c = 2 ; c <= i - 1 ; c++ )
      {
         if ( i%c == 0 )
            break;
      }
      if ( c == i )
      {
         printf("%d\n", i);
         count++;
      }
      i++;
   }
 
   return 0;
}
